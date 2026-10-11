/* podshim: an LD_PRELOAD network shim that gives a pod a fake interface and a fake peer (R66).
 * RCCL, gloo and vLLM advertise an interface address and bind to it; the pods have no interface that reaches each
 * other, only a tunnel that ends on 127.0.0.1. The process sees one more interface (PODSHIM_IFNAME, PODSHIM_SELF/24);
 * a bind to SELF becomes the wildcard, and SELF:0 is pinned into a fixed port range the tunnel carries; a connect to
 * PEER:p goes into the tunnel, one to SELF:p to 127.0.0.1:p; getsockname/getpeername report the fake addresses.
 * IPv4 and IPv4-mapped IPv6 (::ffff:a.b.c.d, which torch's TCPStore client uses) are both handled. Environment:
 *   PODSHIM_SELF, PODSHIM_PEER  fake addresses (SELF unset: the shim does nothing); PODSHIM_IFNAME (default pods0)
 *   PODSHIM_PORT_LOW/HIGH       range for pinned binds (default 29600 to 29727, R66's 128)
 *   PODSHIM_PEER_PORT_LOW       the peer's range start (default PORT_LOW); PODSHIM_PEER_HOST (default 127.0.0.1)
 *   PODSHIM_PEER_PORT_BASE      PEER:p goes to PEER_HOST:(BASE + p - PEER_PORT_LOW) (default PEER_PORT_LOW + 200)
 *   PODSHIM_LOG                 append a one-line trace per rewrite to this file
 * Build: make podshim. Hooks never allocate except getifaddrs and keep per-fd state in a lock-free table. */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define FAKE_INDEX 7919u
#define MAX_FD 65536
#define MAGIC 0x706f647368696d31ULL /* "podshim1" */

static int (*real_getifaddrs)(struct ifaddrs **);
static void (*real_freeifaddrs)(struct ifaddrs *);
static int (*real_bind)(int, const struct sockaddr *, socklen_t);
static int (*real_connect)(int, const struct sockaddr *, socklen_t);
static int (*real_getsockname)(int, struct sockaddr *, socklen_t *);
static int (*real_getpeername)(int, struct sockaddr *, socklen_t *);
static unsigned (*real_if_nametoindex)(const char *);
static char *(*real_if_indextoname)(unsigned, char *);

static pthread_once_t once = PTHREAD_ONCE_INIT;
static int active; /* PODSHIM_SELF parsed */
static struct in_addr self_ip, peer_ip, peer_host;
static char ifname[IF_NAMESIZE];
static unsigned port_low, port_high, peer_low, peer_base, range_len;
static int log_fd = -1;
static atomic_uint next_slot;            /* round-robin start for pinned binds */
static _Atomic uint64_t peer_of[MAX_FD]; /* fd -> 1<<63 | fake ip << 16 | fake port of the redirected connect */

static unsigned env_port(const char *name, unsigned dflt) {
    const char *v = getenv(name);
    long n = v && *v ? strtol(v, NULL, 10) : 0;
    return (n > 0 && n < 65536) ? (unsigned)n : dflt;
}

static int env_ip(const char *name, const char *dflt, struct in_addr *out) {
    const char *v = getenv(name);
    if (!v || !*v) v = dflt;
    return v && inet_pton(AF_INET, v, out) == 1;
}

static void init(void) {
#define REAL(f) real_##f = dlsym(RTLD_NEXT, #f)
    REAL(getifaddrs); REAL(freeifaddrs); REAL(bind); REAL(connect);
    REAL(getsockname); REAL(getpeername); REAL(if_nametoindex); REAL(if_indextoname);
    const char *name = getenv("PODSHIM_IFNAME");
    snprintf(ifname, sizeof ifname, "%s", name && *name ? name : "pods0");
    port_low = env_port("PODSHIM_PORT_LOW", 29600);
    port_high = env_port("PODSHIM_PORT_HIGH", port_low + 127);
    port_high = port_high < port_low ? port_low : port_high, range_len = port_high - port_low + 1;
    peer_low = env_port("PODSHIM_PEER_PORT_LOW", port_low);
    peer_base = env_port("PODSHIM_PEER_PORT_BASE", peer_low + 200);
    if (!env_ip("PODSHIM_SELF", NULL, &self_ip)) return;
    if (!env_ip("PODSHIM_PEER", NULL, &peer_ip)) peer_ip.s_addr = 0;
    if (!env_ip("PODSHIM_PEER_HOST", "127.0.0.1", &peer_host)) peer_host.s_addr = htonl(INADDR_LOOPBACK);
    const char *log = getenv("PODSHIM_LOG");
    if (log && *log) log_fd = open(log, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    active = 1;
}

static inline void ensure(void) { pthread_once(&once, init); }

static void trace(const char *fmt, ...) {
    if (log_fd < 0) return;
    char buf[256];
    int n = snprintf(buf, sizeof buf, "podshim pid %d: ", (int)getpid());
    va_list ap;
    va_start(ap, fmt);
    n += vsnprintf(buf + n, sizeof buf - (size_t)n, fmt, ap);
    va_end(ap);
    if (n > (int)sizeof buf - 2) n = (int)sizeof buf - 2;
    buf[n++] = '\n';
    if (write(log_fd, buf, (size_t)n) < 0) return;
}

/* An AF_INET address, or an AF_INET6 one that carries an IPv4 address (::ffff:a.b.c.d; :: reads as 0.0.0.0). */
static int get_v4(const struct sockaddr *sa, socklen_t len, struct in_addr *ip, unsigned *port) {
    if (!active || !sa) return 0;
    if (sa->sa_family == AF_INET && len >= (socklen_t)sizeof(struct sockaddr_in)) {
        const struct sockaddr_in *s = (const void *)sa;
        *ip = s->sin_addr, *port = ntohs(s->sin_port);
        return 1;
    }
    if (sa->sa_family == AF_INET6 && len >= (socklen_t)sizeof(struct sockaddr_in6)) {
        const struct sockaddr_in6 *s = (const void *)sa;
        if (!IN6_IS_ADDR_V4MAPPED(&s->sin6_addr) && !IN6_IS_ADDR_UNSPECIFIED(&s->sin6_addr)) return 0;
        memcpy(ip, &s->sin6_addr.s6_addr[12], 4), *port = ntohs(s->sin6_port);
        return 1;
    }
    return 0;
}

/* Write ip:port into an address of the same family get_v4 read (IPv6 gets ::ffff:ip, or :: for the wildcard). */
static void set_v4(struct sockaddr *sa, struct in_addr ip, unsigned port) {
    if (sa->sa_family == AF_INET) {
        struct sockaddr_in *s = (void *)sa;
        s->sin_addr = ip, s->sin_port = htons((uint16_t)port);
        return;
    }
    struct sockaddr_in6 *s = (void *)sa;
    memset(&s->sin6_addr, 0, sizeof s->sin6_addr);
    if (ip.s_addr != htonl(INADDR_ANY)) s->sin6_addr.s6_addr[10] = s->sin6_addr.s6_addr[11] = 0xff;
    memcpy(&s->sin6_addr.s6_addr[12], &ip, 4), s->sin6_port = htons((uint16_t)port);
}

/* getifaddrs / freeifaddrs: one extra entry at the tail, recognised by its own layout and a magic word */
struct fake_ifa {
    struct ifaddrs ifa;
    uint64_t magic;
    char name[IF_NAMESIZE];
    struct sockaddr_in addr, mask;
};

static int is_fake(const struct ifaddrs *i) {
    const struct fake_ifa *f = (const struct fake_ifa *)i;
    return i && i->ifa_name == f->name && f->magic == MAGIC;
}

int getifaddrs(struct ifaddrs **ifap) {
    ensure();
    int rc = real_getifaddrs(ifap);
    if (rc != 0 || !active) return rc;
    struct fake_ifa *f = calloc(1, sizeof *f);
    if (!f) return rc; /* the real list is still valid; the fake interface is just missing */
    f->magic = MAGIC;
    memcpy(f->name, ifname, sizeof f->name);
    f->addr.sin_family = f->mask.sin_family = AF_INET;
    f->addr.sin_addr = self_ip;
    f->mask.sin_addr.s_addr = htonl(0xffffff00u);
    f->ifa.ifa_name = f->name;
    f->ifa.ifa_flags = IFF_UP | IFF_RUNNING;
    f->ifa.ifa_addr = (struct sockaddr *)&f->addr;
    f->ifa.ifa_netmask = (struct sockaddr *)&f->mask;
    struct ifaddrs **tail = ifap;
    while (*tail) tail = &(*tail)->ifa_next;
    *tail = &f->ifa;
    return 0;
}

void freeifaddrs(struct ifaddrs *ifa) {
    ensure();
    if (!ifa) return;
    if (is_fake(ifa)) { /* the real list was empty */
        struct ifaddrs *rest = ifa->ifa_next;
        free(ifa);
        if (rest) freeifaddrs(rest);
        return;
    }
    for (struct ifaddrs *i = ifa; i->ifa_next; i = i->ifa_next) {
        if (is_fake(i->ifa_next)) {
            struct ifaddrs *mine = i->ifa_next;
            i->ifa_next = mine->ifa_next; /* NULL: we always append at the tail */
            free(mine);
            break;
        }
    }
    real_freeifaddrs(ifa);
}

unsigned if_nametoindex(const char *name) {
    ensure();
    if (active && name && strcmp(name, ifname) == 0) return FAKE_INDEX;
    return real_if_nametoindex(name);
}

char *if_indextoname(unsigned index, char buf[IF_NAMESIZE]) {
    ensure();
    if (active && index == FAKE_INDEX && buf) return memcpy(buf, ifname, IF_NAMESIZE);
    return real_if_indextoname(index, buf);
}

/* bind: SELF becomes the wildcard; SELF:0 is pinned to the next free port of the range */
int bind(int fd, const struct sockaddr *sa, socklen_t len) {
    ensure();
    struct in_addr ip;
    unsigned port;
    struct sockaddr_storage ss;
    if (!get_v4(sa, len, &ip, &port) || ip.s_addr != self_ip.s_addr || len > sizeof ss) return real_bind(fd, sa, len);
    memcpy(&ss, sa, len);
    struct in_addr any = {htonl(INADDR_ANY)};
    if (port != 0) {
        set_v4((struct sockaddr *)&ss, any, port);
        int rc = real_bind(fd, (struct sockaddr *)&ss, len);
        trace("bind fd %d self:%u -> any:%u rc %d", fd, port, port, rc);
        return rc;
    }
    unsigned start = atomic_fetch_add(&next_slot, 1);
    int saved = errno;
    for (unsigned k = 0; k < range_len; k++) {
        port = port_low + (start + k) % range_len;
        set_v4((struct sockaddr *)&ss, any, port);
        if (real_bind(fd, (struct sockaddr *)&ss, len) == 0) {
            atomic_fetch_add(&next_slot, k); /* skip the ports we found busy */
            errno = saved;
            trace("bind fd %d self:0 -> any:%u (pinned)", fd, port);
            return 0;
        }
        if (errno != EADDRINUSE && errno != EACCES) return -1;
    }
    trace("bind fd %d self:0 -> range %u-%u full", fd, port_low, port_high);
    errno = EADDRINUSE;
    return -1;
}

/* connect: PEER:p goes to PEER_HOST:(BASE + p - PEER_LOW); SELF:p goes to 127.0.0.1:p */
static void remember(int fd, uint64_t v) {
    if (fd >= 0 && fd < MAX_FD) atomic_store(&peer_of[fd], v);
}

int connect(int fd, const struct sockaddr *sa, socklen_t len) {
    ensure();
    struct in_addr ip, to;
    unsigned port, to_port;
    struct sockaddr_storage ss;
    if (!get_v4(sa, len, &ip, &port) || len > sizeof ss) { remember(fd, 0); return real_connect(fd, sa, len); }
    if (peer_ip.s_addr && ip.s_addr == peer_ip.s_addr && port >= peer_low && port < peer_low + range_len) {
        to = peer_host, to_port = peer_base + port - peer_low;
    } else if (ip.s_addr == self_ip.s_addr) {
        to.s_addr = htonl(INADDR_LOOPBACK), to_port = port;
    } else {
        if (ip.s_addr == peer_ip.s_addr) trace("connect fd %d peer:%u outside the peer range, left alone", fd, port);
        remember(fd, 0);
        return real_connect(fd, sa, len);
    }
    remember(fd, (1ULL << 63) | ((uint64_t)ntohl(ip.s_addr) << 16) | port);
    memcpy(&ss, sa, len);
    set_v4((struct sockaddr *)&ss, to, to_port);
    int rc = real_connect(fd, (struct sockaddr *)&ss, len), e = errno;
    char a[INET_ADDRSTRLEN], b[INET_ADDRSTRLEN];
    trace("connect fd %d %s:%u -> %s:%u family %d rc %d errno %d", fd, inet_ntop(AF_INET, &ip, a, sizeof a), port,
          inet_ntop(AF_INET, &to, b, sizeof b), to_port, sa->sa_family, rc, rc ? e : 0);
    errno = e;
    return rc;
}

/* getsockname / getpeername report the fake addresses. mapped_peer: the fake peer a connect on fd was redirected
 * from, while the socket is still connected where we sent it. */
static int mapped_peer(int fd, struct in_addr *ip, unsigned *port) {
    if (fd < 0 || fd >= MAX_FD) return 0;
    uint64_t v = atomic_load(&peer_of[fd]);
    if (!(v >> 63)) return 0;
    struct sockaddr_storage real;
    socklen_t rl = sizeof real;
    struct in_addr rip;
    unsigned rport;
    if (real_getpeername(fd, (struct sockaddr *)&real, &rl) != 0 || !get_v4((struct sockaddr *)&real, rl, &rip, &rport))
        return 0;
    ip->s_addr = htonl((uint32_t)(v >> 16)), *port = (unsigned)(v & 0xffff);
    unsigned want = ip->s_addr == self_ip.s_addr ? *port : peer_base + *port - peer_low;
    return rport == want; /* a stale entry from a closed fd that was reused does not match */
}

int getsockname(int fd, struct sockaddr *sa, socklen_t *len) {
    ensure();
    socklen_t cap = len ? *len : 0;
    int rc = real_getsockname(fd, sa, len);
    struct in_addr ip, pip;
    unsigned port, pport;
    if (rc != 0 || *len > cap || !get_v4(sa, *len, &ip, &port)) return rc;
    uint32_t h = ntohl(ip.s_addr);
    int local = h == INADDR_ANY || (h >> 24) == 127;
    if (local && ((port >= port_low && port <= port_high) || mapped_peer(fd, &pip, &pport)))
        set_v4(sa, self_ip, port);
    return rc;
}

int getpeername(int fd, struct sockaddr *sa, socklen_t *len) {
    ensure();
    socklen_t cap = len ? *len : 0;
    int rc = real_getpeername(fd, sa, len);
    struct in_addr ip, fake;
    unsigned port, fport;
    if (rc == 0 && *len <= cap && get_v4(sa, *len, &ip, &port) && mapped_peer(fd, &fake, &fport))
        set_v4(sa, fake, fport);
    return rc;
}
