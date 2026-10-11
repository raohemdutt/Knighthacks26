-- Circle shared state in Neon Postgres (N1). Safe to re-run.
--   psql "$NEON_DATABASE_URL" -f contexts/circle/infrastructure/schema.sql
--
-- Only approved summaries, proposals, counts and scores live here. Frames, raw
-- habit signals, connector items and face templates never leave the box.
-- (One exception, off by default: sealed_messages at the end of this file holds a
-- member's own face template as ciphertext only the one friend's box it is
-- addressed to can read. Nothing in this database can read a face from it.)

CREATE TABLE IF NOT EXISTS circles (
    id          text PRIMARY KEY,
    name        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS members (
    id            text PRIMARY KEY,
    circle_id     text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    display_name  text NOT NULL,
    has_box       boolean NOT NULL DEFAULT false,  -- false = phone-only member
    joined_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS members_circle ON members (circle_id);

-- Every approved share lands here first, keyed by the box's share_id, so the
-- outbox worker can retry without duplicating (contracts.circle.ShareQueued).
CREATE TABLE IF NOT EXISTS shares (
    share_id    text PRIMARY KEY,
    circle_id   text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    member_id   text NOT NULL REFERENCES members(id) ON DELETE CASCADE,
    kind        text NOT NULL CHECK (kind IN ('event_proposal', 'reply', 'habit_summary', 'impact_count', 'game_score')),
    payload     jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS shares_circle_time ON shares (circle_id, created_at);

-- Agent-to-agent thread for one proposal (contracts.circle.MessageReceived).
CREATE TABLE IF NOT EXISTS agent_messages (
    id           text PRIMARY KEY,
    circle_id    text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    thread_id    text NOT NULL,
    from_member  text NOT NULL REFERENCES members(id) ON DELETE CASCADE,
    kind         text NOT NULL CHECK (kind IN ('proposal', 'reply')),
    proposal     jsonb,
    reply        jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    CHECK ((kind = 'proposal' AND proposal IS NOT NULL) OR (kind = 'reply' AND reply IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS agent_messages_thread ON agent_messages (circle_id, thread_id, created_at);
-- Every box reads "this circle's messages since <time>" on each tick; the thread index cannot serve that.
CREATE INDEX IF NOT EXISTS agent_messages_circle_time ON agent_messages (circle_id, created_at);

-- Shared calendar: an event exists once a proposal is agreed (contracts.circle.EventAgreed).
CREATE TABLE IF NOT EXISTS events (
    id          text PRIMARY KEY,
    circle_id   text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    thread_id   text UNIQUE,
    title       text NOT NULL,
    starts_at   timestamptz NOT NULL,
    ends_at     timestamptz NOT NULL,
    location    text,
    created_by  text NOT NULL REFERENCES members(id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    CHECK (ends_at > starts_at)
);
CREATE INDEX IF NOT EXISTS events_circle_start ON events (circle_id, starts_at);

CREATE TABLE IF NOT EXISTS rsvps (
    event_id    text NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    member_id   text NOT NULL REFERENCES members(id) ON DELETE CASCADE,
    answer      text NOT NULL CHECK (answer IN ('free', 'busy', 'counter')),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, member_id)
);

-- One row per member, habit and day: the daily summary only, never raw signals
-- (contracts.habits.DayClosed).
CREATE TABLE IF NOT EXISTS habit_summaries (
    member_id   text NOT NULL REFERENCES members(id) ON DELETE CASCADE,
    circle_id   text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    habit_id    text NOT NULL,
    day         date NOT NULL,
    met         boolean NOT NULL,
    value       double precision NOT NULL,
    streak      integer NOT NULL DEFAULT 0,
    PRIMARY KEY (member_id, habit_id, day)
);
CREATE INDEX IF NOT EXISTS habit_summaries_circle_day ON habit_summaries (circle_id, day);

-- Latest Waste Watch totals per member (contracts.impact.CountUpdated).
CREATE TABLE IF NOT EXISTS impact_counters (
    member_id        text PRIMARY KEY REFERENCES members(id) ON DELETE CASCADE,
    circle_id        text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    items_seen       integer NOT NULL DEFAULT 0,
    recyclable_seen  integer NOT NULL DEFAULT 0,
    recycled         integer NOT NULL DEFAULT 0,
    contamination    integer NOT NULL DEFAULT 0,
    diversion_rate   double precision NOT NULL DEFAULT 0,
    updated_at       timestamptz NOT NULL DEFAULT now()
);

-- One row per scored round (contracts.play.RoundScored).
CREATE TABLE IF NOT EXISTS game_scores (
    round_id    text PRIMARY KEY,
    session_id  text NOT NULL,
    member_id   text NOT NULL REFERENCES members(id) ON DELETE CASCADE,
    circle_id   text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    item        text NOT NULL,
    players     integer NOT NULL DEFAULT 1,
    correct     boolean NOT NULL,
    seconds     double precision NOT NULL,
    path_len    integer NOT NULL DEFAULT 0,
    score       double precision NOT NULL,
    played_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS game_scores_circle_score ON game_scores (circle_id, score DESC);

-- Opt-in coarse place a member chose to share (G5): a named spot, never
-- coordinates, and it expires.
CREATE TABLE IF NOT EXISTS presence (
    member_id   text PRIMARY KEY REFERENCES members(id) ON DELETE CASCADE,
    circle_id   text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    place       text NOT NULL,
    source      text NOT NULL CHECK (source IN ('phone', 'desk')),
    shared_at   timestamptz NOT NULL DEFAULT now(),
    expires_at  timestamptz NOT NULL
);
-- Every box reads "this circle's unexpired places" on each tick.
CREATE INDEX IF NOT EXISTS presence_circle_expiry ON presence (circle_id, expires_at);

-- The League (contexts/play/league): the stake per week, each declared week's standings, and the
-- dares. Standings are the week's summary lines only (contracts.play.Standing): never raw signals.
CREATE TABLE IF NOT EXISTS league_stakes (
    circle_id   text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    week_start  date NOT NULL,
    kind        text NOT NULL CHECK (kind IN ('pays_for_food', 'picks_the_event', 'custom')),
    text        text,
    set_by      text NOT NULL REFERENCES members(id) ON DELETE CASCADE,
    set_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (circle_id, week_start)
);

CREATE TABLE IF NOT EXISTS league_weeks (
    circle_id   text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    week_start  date NOT NULL,
    week_end    date NOT NULL,
    standings   jsonb NOT NULL,
    worst       text,
    best        text,
    stake       jsonb NOT NULL,
    summary     text NOT NULL,
    closed_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (circle_id, week_start)
);

CREATE TABLE IF NOT EXISTS league_challenges (
    challenge_id  text PRIMARY KEY,
    circle_id     text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    title         text NOT NULL,
    habit_id      text NOT NULL,
    direction     text NOT NULL CHECK (direction IN ('at_least', 'at_most')),
    target        double precision NOT NULL,
    starts        date NOT NULL,
    ends          date NOT NULL,
    members       jsonb NOT NULL,
    started_by    text NOT NULL REFERENCES members(id) ON DELETE CASCADE,
    results       jsonb,
    winners       jsonb,
    summary       text,
    closed_at     timestamptz,
    CHECK (ends >= starts)
);
CREATE INDEX IF NOT EXISTS league_challenges_circle ON league_challenges (circle_id, starts);

CREATE OR REPLACE VIEW habit_leaderboard AS
SELECT h.circle_id, h.member_id, m.display_name, h.habit_id, h.streak, h.day
FROM habit_summaries h
JOIN members m ON m.id = h.member_id
WHERE h.day = (SELECT max(day) FROM habit_summaries x WHERE x.member_id = h.member_id AND x.habit_id = h.habit_id);

CREATE OR REPLACE VIEW impact_leaderboard AS
SELECT i.circle_id, i.member_id, m.display_name, i.items_seen, i.recycled, i.contamination, i.diversion_rate
FROM impact_counters i
JOIN members m ON m.id = i.member_id;

-- The sealed channel (face share, off by default; contracts.circle.SealedMessage, version 1).
-- box_key: a box's two public keys, self-signed (to_member is NULL). face_share: a signed
-- header and ciphertext readable only by to_member's box. face_revoke: a signed header.
-- `kind` has no CHECK on purpose: a box skips kinds and versions it does not know, so
-- later kinds stay additive. Rows are small and deleted by the boxes when used or revoked.
-- expires_at is the expiry in a face_share's signed header, copied out so the database side can act on it
-- without reading the body: the inbox leaves out a row that is past it and the sender's box sweeps it.
-- NULL: the row has no expiry of its own (a key, a revoke) or a box from before this column wrote it.
CREATE TABLE IF NOT EXISTS sealed_messages (
    id           text PRIMARY KEY,
    v            integer NOT NULL DEFAULT 1,
    circle_id    text NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
    from_member  text NOT NULL REFERENCES members(id) ON DELETE CASCADE,
    to_member    text,
    kind         text NOT NULL,
    body         jsonb NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz,
    CHECK (pg_column_size(body) <= 400000)
);
-- A database created before expires_at gets it here: CREATE TABLE IF NOT EXISTS does not add a column.
-- Nullable with no default, so adding it rewrites nothing and existing rows read NULL.
ALTER TABLE sealed_messages ADD COLUMN IF NOT EXISTS expires_at timestamptz;

CREATE INDEX IF NOT EXISTS sealed_messages_inbox ON sealed_messages (circle_id, to_member, created_at);
