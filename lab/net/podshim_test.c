/* podshim_test: proves lab/net/podshim.so in one process (R66). Run with the shim preloaded and its environment set:
 *   LD_PRELOAD=./podshim.so PODSHIM_SELF=10.99.0.1 PODSHIM_PEER=10.99.0.2 PODSHIM_PORT_LOW=40000 \
 *   PODSHIM_PEER_PORT_LOW=41000 PODSHIM_PEER_PORT_BASE=42000 ./podshim_test
 * The "tunnel" is a listener on 127.0.0.1:(PEER_PORT_BASE + 5); a connect to PEER:(PEER_PORT_LOW + 5) must reach it.
 * Prints one line per check ("ok ..." or "FAIL ...") and the round-trip time; exits 0 only when every check passed. */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int failures;
static struct in_addr self_ip, peer_ip;
static unsigned port_low, port_n, peer_low, peer_base;
static const char *ifname;

#define CHECK(cond, ...) do { int c_ = (cond); printf(c_ ? "ok   " : "FAIL "); printf(__VA_ARGS__); printf("\n"); \
                              if (!c_) failures++; } while (0)

static unsigned env_u(const char *k, unsigned d) { const char *v = getenv(k); return v && *v ? (unsigned)atoi(v) : d; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

static struct sockaddr_in sin_of(struct in_addr a, unsigned port) {
    struct sockaddr_in s = {0};
    s.sin_family = AF_INET; s.sin_addr = a; s.sin_port = htons((unsigned short)port);
    return s;
}

static int tcp(void) { return socket(AF_INET, SOCK_STREAM, 0); }

static int tunnel_listener(unsigned port) {
    int fd = tcp(), one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct in_addr lo; inet_pton(AF_INET, "127.0.0.1", &lo);
    struct sockaddr_in s = sin_of(lo, port);
    if (bind(fd, (struct sockaddr *)&s, sizeof s) || listen(fd, 128)) { perror("tunnel listener"); exit(2); }
    return fd;
}

static void *echo_server(void *arg) { /* accept forever; echo each byte back */
    int lfd = (int)(long)arg;
    for (;;) {
        int c = accept(lfd, NULL, NULL);
        if (c < 0) return NULL;
        char b[64]; ssize_t n;
        while ((n = read(c, b, sizeof b)) > 0) if (write(c, b, (size_t)n) != n) break;
        close(c);
    }
}

static int connect_peer(unsigned port) {
    int fd = tcp();
    struct sockaddr_in s = sin_of(peer_ip, port);
    if (connect(fd, (struct sockaddr *)&s, sizeof s)) { close(fd); return -1; }
    return fd;
}

static int peer_is(int fd, struct in_addr a, unsigned port) {
    struct sockaddr_in s; socklen_t l = sizeof s;
    return getpeername(fd, (struct sockaddr *)&s, &l) == 0 && s.sin_addr.s_addr == a.s_addr && ntohs(s.sin_port) == port;
}

static int local_is_self(int fd, unsigned *port) {
    struct sockaddr_in s; socklen_t l = sizeof s;
    if (getsockname(fd, (struct sockaddr *)&s, &l)) return 0;
    if (port) *port = ntohs(s.sin_port);
    return s.sin_addr.s_addr == self_ip.s_addr;
}

static void *hammer(void *arg) { /* one thread's share of concurrent connects through the shim */
    long bad = 0;
    for (int i = 0; i < 50; i++) {
        int fd = connect_peer(peer_low + 5);
        char b = 'x';
        if (fd < 0 || !peer_is(fd, peer_ip, peer_low + 5) || write(fd, &b, 1) != 1 || read(fd, &b, 1) != 1) bad++;
        if (fd >= 0) close(fd);
    }
    *(long *)arg = bad;
    return NULL;
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (inet_pton(AF_INET, getenv("PODSHIM_SELF") ? getenv("PODSHIM_SELF") : "", &self_ip) != 1 ||
        inet_pton(AF_INET, getenv("PODSHIM_PEER") ? getenv("PODSHIM_PEER") : "", &peer_ip) != 1) {
        fprintf(stderr, "set PODSHIM_SELF and PODSHIM_PEER\n"); return 2;
    }
    ifname = getenv("PODSHIM_IFNAME") ? getenv("PODSHIM_IFNAME") : "pods0";
    port_low = env_u("PODSHIM_PORT_LOW", 29600);
    port_n = env_u("PODSHIM_PORT_HIGH", port_low + 127) - port_low + 1;
    peer_low = env_u("PODSHIM_PEER_PORT_LOW", port_low);
    peer_base = env_u("PODSHIM_PEER_PORT_BASE", peer_low + 200);
    if (port_n < 1 || port_n > 1024) { fprintf(stderr, "range too large for the test\n"); return 2; }

    /* 1. the fake interface */
    struct ifaddrs *ifs, *i;
    int found = 0;
    if (getifaddrs(&ifs) == 0) {
        for (i = ifs; i; i = i->ifa_next) {
            char a[INET6_ADDRSTRLEN] = "-";
            if (i->ifa_addr && i->ifa_addr->sa_family == AF_INET)
                inet_ntop(AF_INET, &((struct sockaddr_in *)i->ifa_addr)->sin_addr, a, sizeof a);
            printf("     iface %-8s family %d addr %s flags 0x%x\n", i->ifa_name,
                   i->ifa_addr ? i->ifa_addr->sa_family : -1, a, i->ifa_flags);
            if (strcmp(i->ifa_name, ifname) == 0 && i->ifa_addr && i->ifa_addr->sa_family == AF_INET &&
                ((struct sockaddr_in *)i->ifa_addr)->sin_addr.s_addr == self_ip.s_addr &&
                ((struct sockaddr_in *)i->ifa_netmask)->sin_addr.s_addr == htonl(0xffffff00u) &&
                (i->ifa_flags & (IFF_UP | IFF_RUNNING)) == (IFF_UP | IFF_RUNNING)) found = 1;
        }
        freeifaddrs(ifs);
    }
    CHECK(found, "getifaddrs lists %s with the self address, /24, UP RUNNING", ifname);
    for (int k = 0; k < 2000; k++) if (getifaddrs(&ifs) == 0) freeifaddrs(ifs);
    CHECK(1, "2000 getifaddrs/freeifaddrs rounds");
    unsigned idx = if_nametoindex(ifname);
    char nb[IF_NAMESIZE] = "";
    CHECK(idx != 0 && if_indextoname(idx, nb) && strcmp(nb, ifname) == 0, "if_nametoindex %u and back to %s", idx, nb);
    CHECK(if_nametoindex("lo") != 0 && if_nametoindex("lo") != idx, "if_nametoindex(lo) still the real one");

    /* 2. bind to the fake address, port 0: pinned into the range, reported as SELF */
    int l = tcp();
    struct sockaddr_in s = sin_of(self_ip, 0);
    unsigned lport = 0;
    CHECK(bind(l, (struct sockaddr *)&s, sizeof s) == 0 && listen(l, 16) == 0, "bind SELF:0 and listen");
    CHECK(local_is_self(l, &lport) && lport >= port_low && lport < port_low + port_n,
          "getsockname reports SELF:%u (range %u-%u)", lport, port_low, port_low + port_n - 1);
    int c = tcp();
    s = sin_of(self_ip, lport);
    int a = (connect(c, (struct sockaddr *)&s, sizeof s) == 0) ? accept(l, NULL, NULL) : -1;
    CHECK(a >= 0 && peer_is(c, self_ip, lport), "connect SELF:%u reaches our own listener; getpeername says SELF", lport);
    close(c); close(a); close(l); /* the client closes first, so no range port is left in TIME_WAIT */

    /* 3. the range holds port_n ports; one more pinned bind fails cleanly */
    int fds[1025], got = 0, distinct = 1, want = (int)port_n;
    unsigned seen[1025];
    for (int k = 0; k <= want; k++) {
        fds[k] = tcp();
        s = sin_of(self_ip, 0);
        if (bind(fds[k], (struct sockaddr *)&s, sizeof s) == 0) {
            local_is_self(fds[k], &seen[got]);
            for (int j = 0; j < got; j++) if (seen[j] == seen[got]) distinct = 0;
            got++;
        }
    }
    /* every free range port once, then refused; a port still in TIME_WAIT from an earlier run counts as busy */
    CHECK(got > want - 8 && got <= want && distinct, "%d of %d range ports pinned, all distinct, then refused", got, want);
    for (int k = 0; k <= want; k++) close(fds[k]);

    /* 4. connect PEER:(PEER_LOW+5) lands on the tunnel end 127.0.0.1:(PEER_BASE+5) */
    int t = tunnel_listener(peer_base + 5);
    pthread_t srv;
    pthread_create(&srv, NULL, echo_server, (void *)(long)t);
    c = connect_peer(peer_low + 5);
    CHECK(c >= 0, "connect PEER:%u -> 127.0.0.1:%u", peer_low + 5, peer_base + 5);
    CHECK(c >= 0 && peer_is(c, peer_ip, peer_low + 5), "getpeername reports PEER:%u", peer_low + 5);
    CHECK(c >= 0 && local_is_self(c, NULL), "getsockname on the connected socket reports SELF");
    int one = 1;
    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    char b = 'p';
    int n = 2000, okrt = 1;
    double t0 = now();
    for (int k = 0; k < n && okrt; k++) okrt = write(c, &b, 1) == 1 && read(c, &b, 1) == 1;
    double rt = (now() - t0) / n * 1e6;
    CHECK(okrt, "%d one-byte round trips through the mapping, %.1f us each", n, rt);
    close(c);

    /* 4b. the same through an IPv6 socket to ::ffff:PEER (torch's TCPStore client connects this way) */
    c = socket(AF_INET6, SOCK_STREAM, 0);
    struct sockaddr_in6 s6 = {0}, p6;
    socklen_t l6 = sizeof p6;
    s6.sin6_family = AF_INET6;
    s6.sin6_port = htons((unsigned short)(peer_low + 5));
    s6.sin6_addr.s6_addr[10] = s6.sin6_addr.s6_addr[11] = 0xff;
    memcpy(&s6.sin6_addr.s6_addr[12], &peer_ip, 4);
    CHECK(connect(c, (struct sockaddr *)&s6, sizeof s6) == 0 && write(c, &b, 1) == 1 && read(c, &b, 1) == 1 &&
          getpeername(c, (struct sockaddr *)&p6, &l6) == 0 && memcmp(&p6.sin6_addr.s6_addr[12], &peer_ip, 4) == 0 &&
          ntohs(p6.sin6_port) == peer_low + 5, "connect ::ffff:PEER:%u through an IPv6 socket, getpeername says so", peer_low + 5);
    close(c);

    /* 5. eight threads, 50 connects each, at the same time */
    pthread_t th[8];
    long bad[8];
    t0 = now();
    for (int k = 0; k < 8; k++) pthread_create(&th[k], NULL, hammer, &bad[k]);
    long badsum = 0;
    for (int k = 0; k < 8; k++) { pthread_join(th[k], NULL); badsum += bad[k]; }
    CHECK(badsum == 0, "8 threads x 50 concurrent connects through the shim, %ld bad, %.3f s", badsum, now() - t0);

    /* 6. other addresses pass through untouched */
    struct in_addr lo; inet_pton(AF_INET, "127.0.0.1", &lo);
    l = tcp();
    s = sin_of(lo, 0);
    struct sockaddr_in r; socklen_t rl = sizeof r;
    CHECK(bind(l, (struct sockaddr *)&s, sizeof s) == 0 && getsockname(l, (struct sockaddr *)&r, &rl) == 0 &&
          r.sin_addr.s_addr == lo.s_addr, "bind 127.0.0.1:0 left alone (port %u)", ntohs(r.sin_port));
    close(l);

    printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
