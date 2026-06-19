import heapq
import math
import random
import time
from typing import override

from gamelib.hex.agent import Agent
from gamelib.hex.gamestate import GameState as State
from gamelib.hex.move import Move


class HexAgent(Agent):
    """A fast Hex agent built around shortest-path pressure."""

    # Hex-grid neighbor offsets for the row/column coordinates used by gamelib.
    NEIGHBORS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
    BRIDGES = (
        (-2, 1, ((-1, 0), (-1, 1))),
        (-1, -1, ((-1, 0), (0, -1))),
        (-1, 2, ((-1, 1), (0, 1))),
        (1, -2, ((0, -1), (1, -1))),
        (1, 1, ((0, 1), (1, 0))),
        (2, -1, ((1, -1), (1, 0))),
    )

    @override
    def initialize(self, init_data: dict) -> None:
        self.player_id = init_data["player_id"]
        self.rng = random.Random(20260613 + self.player_id)

    @override
    def get_move(self, state: State) -> Move:
        start = time.perf_counter()
        board = state.board
        size = state.board_size
        me = self.player_id
        opp = 1 - me
        empty = [(r, c) for r in range(size) for c in range(size) if board[r][c] == -1]
        if not empty:
            raise ValueError("No valid moves available.")

        if len(empty) == size * size:
            return Move(player=me, position=[4, 4])

        # Center is a flexible opening because it can grow toward either side.
        if board[size // 2][size // 2] == -1:
            return Move(player=me, position=[size // 2, size // 2])

        # Tactical layer: win immediately, otherwise stop an immediate loss.
        winning = self._moves_that_complete_connection(board, size, empty, me)
        if winning:
            return Move(player=me, position=list(self._best_positional(winning, size, me)))

        blocking = self._moves_that_complete_connection(board, size, empty, opp)
        if blocking:
            return Move(player=me, position=list(self._best_positional(blocking, size, me)))

        my_now = self._connection_cost(board, size, me)
        opp_now = self._connection_cost(board, size, opp)

        # Most useful Hex moves are near existing stones; this keeps scoring fast.
        scored = []
        candidates = self._candidate_moves(board, size, empty)
        if not candidates:
            candidates = empty
        candidates.sort(key=lambda move: self._quick_prior(board, size, move[0], move[1], me, opp), reverse=True)
        candidates = candidates[:36]

        for move in candidates:
            if time.perf_counter() - start > 3.35:
                break
            r, c = move
            board[r][c] = me
            my_after = self._connection_cost(board, size, me)
            board[r][c] = opp
            opp_after = self._connection_cost(board, size, opp)
            board[r][c] = -1

            # Main idea: shorten our cheapest connection and lengthen theirs.
            own_gain = my_now - my_after
            opp_damage = opp_after - opp_now
            score = 125.0 * own_gain + 65.0 * opp_damage
            score += self._shape_score(board, size, r, c, me, opp)
            score += self._centrality(size, r, c) * 2.0
            score += self.rng.random() * 0.001
            scored.append((score, move))

        if not scored:
            move = self._best_positional(empty, size, me)
        else:
            move = self._refine_top_moves(board, size, scored, me, opp, start)

        return Move(player=me, position=list(move))

    def _refine_top_moves(
        self,
        board: list[list[int]],
        size: int,
        scored: list[tuple[float, tuple[int, int]]],
        me: int,
        opp: int,
        start: float,
    ) -> tuple[int, int]:
        scored.sort(reverse=True)
        best_move = scored[0][1]
        refined = []

        for first_score, move in scored[:6]:
            if time.perf_counter() - start > 2.75:
                break

            r, c = move
            board[r][c] = me
            if self._has_connection(board, size, me):
                reply_score = 1_000_000.0
            else:
                reply_score = self._worst_reply_score(board, size, me, opp, start)
            board[r][c] = -1

            refined.append((reply_score * 0.82 + first_score * 0.16, move))

        if refined:
            refined.sort(reverse=True)
            best_move = refined[0][1]
        return best_move

    def _worst_reply_score(
        self, board: list[list[int]], size: int, me: int, opp: int, start: float
    ) -> float:
        empty = [(r, c) for r in range(size) for c in range(size) if board[r][c] == -1]
        if not empty:
            return self._position_score(board, size, me, opp)

        reply_candidates = self._candidate_moves(board, size, empty)
        if not reply_candidates:
            reply_candidates = empty
        reply_candidates.sort(key=lambda move: self._quick_prior(board, size, move[0], move[1], opp, me), reverse=True)

        worst_for_us = math.inf
        for r, c in reply_candidates[:12]:
            if time.perf_counter() - start > 3.45:
                break
            board[r][c] = opp
            worst_for_us = min(worst_for_us, self._position_score(board, size, me, opp))
            board[r][c] = -1

        if worst_for_us == math.inf:
            return self._position_score(board, size, me, opp)
        return worst_for_us

    def _position_score(self, board: list[list[int]], size: int, me: int, opp: int) -> float:
        if self._has_connection(board, size, me):
            return 1_000_000.0
        if self._has_connection(board, size, opp):
            return -1_000_000.0

        my_cost = self._connection_cost(board, size, me)
        opp_cost = self._connection_cost(board, size, opp)
        if my_cost == math.inf and opp_cost == math.inf:
            return 0.0
        if my_cost == math.inf:
            return -10_000.0
        if opp_cost == math.inf:
            return 10_000.0
        return (opp_cost - my_cost) * 130.0

    def _quick_prior(self, board: list[list[int]], size: int, r: int, c: int, me: int, opp: int) -> float:
        own_neighbors = 0
        opp_neighbors = 0
        for nr, nc in self._neighbors(size, r, c):
            if board[nr][nc] == me:
                own_neighbors += 1
            elif board[nr][nc] == opp:
                opp_neighbors += 1
        return (
            self._centrality(size, r, c)
            + own_neighbors * 3.0
            + opp_neighbors * 1.4
            + self._bridge_creation_score(board, size, r, c, me, opp) * 0.7
            + self._bridge_block_score(board, size, r, c, opp, me) * 0.4
        )

    def _candidate_moves(self, board: list[list[int]], size: int, empty: list[tuple[int, int]]) -> list[tuple[int, int]]:
        occupied = [(r, c) for r in range(size) for c in range(size) if board[r][c] != -1]
        if not occupied:
            return empty
        seen = set()
        for r, c in occupied:
            for dr, dc in self.NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == -1:
                    seen.add((nr, nc))
            for dr, dc, _common in self.BRIDGES:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == -1:
                    seen.add((nr, nc))
        return list(seen)

    def _moves_that_complete_connection(
        self, board: list[list[int]], size: int, empty: list[tuple[int, int]], player: int
    ) -> list[tuple[int, int]]:
        wins = []
        for r, c in empty:
            board[r][c] = player
            if self._has_connection(board, size, player):
                wins.append((r, c))
            board[r][c] = -1
        return wins

    def _has_connection(self, board: list[list[int]], size: int, player: int) -> bool:
        stack = []
        seen = set()
        if player == 0:
            for r in range(size):
                if board[r][0] == player:
                    stack.append((r, 0))
                    seen.add((r, 0))
            target = lambda _r, c: c == size - 1
        else:
            for c in range(size):
                if board[0][c] == player:
                    stack.append((0, c))
                    seen.add((0, c))
            target = lambda r, _c: r == size - 1

        while stack:
            r, c = stack.pop()
            if target(r, c):
                return True
            for nr, nc in self._neighbors(size, r, c):
                if (nr, nc) not in seen and board[nr][nc] == player:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return False

    def _connection_cost(self, board: list[list[int]], size: int, player: int) -> float:
        # Dijkstra over the board: own stones cost 0, empty cells cost 1,
        # opponent stones are blocked. Lower cost means closer to connecting.
        dist = [[math.inf] * size for _ in range(size)]
        heap = []

        if player == 0:
            for r in range(size):
                cost = self._cell_cost(board[r][0], player)
                if cost < math.inf:
                    dist[r][0] = cost
                    heapq.heappush(heap, (cost, r, 0))
            target = lambda _r, c: c == size - 1
        else:
            for c in range(size):
                cost = self._cell_cost(board[0][c], player)
                if cost < math.inf:
                    dist[0][c] = cost
                    heapq.heappush(heap, (cost, 0, c))
            target = lambda r, _c: r == size - 1

        while heap:
            cost, r, c = heapq.heappop(heap)
            if cost != dist[r][c]:
                continue
            if target(r, c):
                return cost
            for nr, nc in self._neighbors(size, r, c):
                step = self._cell_cost(board[nr][nc], player)
                new_cost = cost + step
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    heapq.heappush(heap, (new_cost, nr, nc))
            if board[r][c] == player:
                for nr, nc in self._virtual_bridge_neighbors(board, size, r, c, player):
                    if cost < dist[nr][nc]:
                        dist[nr][nc] = cost
                        heapq.heappush(heap, (cost, nr, nc))
        return math.inf

    def _cell_cost(self, value: int, player: int) -> float:
        if value == player:
            return 0.0
        if value == -1:
            return 1.0
        return math.inf

    def _shape_score(self, board: list[list[int]], size: int, r: int, c: int, me: int, opp: int) -> float:
        # Small tie-breakers for locally sensible Hex shapes.
        own_neighbors = 0
        opp_neighbors = 0
        empty_neighbors = 0
        for nr, nc in self._neighbors(size, r, c):
            if board[nr][nc] == me:
                own_neighbors += 1
            elif board[nr][nc] == opp:
                opp_neighbors += 1
            else:
                empty_neighbors += 1

        score = own_neighbors * 5.0 + opp_neighbors * 2.5 + empty_neighbors * 0.4
        score += self._bridge_creation_score(board, size, r, c, me, opp)
        score += self._bridge_block_score(board, size, r, c, opp, me)
        if me == 0:
            score += (size - 1 - abs((size - 1) - 2 * c)) * 0.45
        else:
            score += (size - 1 - abs((size - 1) - 2 * r)) * 0.45
        return score

    def _best_positional(self, moves: list[tuple[int, int]], size: int, player: int) -> tuple[int, int]:
        return max(moves, key=lambda move: (self._centrality(size, *move), self._axis_progress(size, *move, player)))

    def _centrality(self, size: int, r: int, c: int) -> float:
        center = (size - 1) / 2
        return size - abs(r - center) - abs(c - center) * 0.85

    def _axis_progress(self, size: int, r: int, c: int, player: int) -> float:
        return min(c, size - 1 - c) if player == 0 else min(r, size - 1 - r)

    def _virtual_bridge_neighbors(
        self, board: list[list[int]], size: int, r: int, c: int, player: int
    ):
        for dr, dc, common in self.BRIDGES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == player:
                if self._bridge_is_safe(board, r, c, player, common):
                    yield nr, nc

    def _bridge_is_safe(
        self, board: list[list[int]], r: int, c: int, player: int, common: tuple[tuple[int, int], tuple[int, int]]
    ) -> bool:
        opp = 1 - player
        return all(board[r + dr][c + dc] != opp for dr, dc in common)

    def _bridge_creation_score(
        self, board: list[list[int]], size: int, r: int, c: int, me: int, opp: int
    ) -> float:
        score = 0.0
        for dr, dc, common in self.BRIDGES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == me:
                if all(board[r + cr][c + cc] != opp for cr, cc in common):
                    score += 8.0
        return score

    def _bridge_block_score(
        self, board: list[list[int]], size: int, r: int, c: int, opp: int, me: int
    ) -> float:
        score = 0.0
        for ar, ac in self._neighbors(size, r, c):
            if board[ar][ac] != opp:
                continue
            for dr, dc, common in self.BRIDGES:
                br, bc = ar + dr, ac + dc
                if not (0 <= br < size and 0 <= bc < size and board[br][bc] == opp):
                    continue
                cells = {(ar + cr, ac + cc) for cr, cc in common}
                if (r, c) in cells and all(board[x][y] != me for x, y in cells):
                    score += 4.0
        return score

    def _neighbors(self, size: int, r: int, c: int):
        for dr, dc in self.NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                yield nr, nc


if __name__ == "__main__":
    agent = HexAgent()
    agent.start()
