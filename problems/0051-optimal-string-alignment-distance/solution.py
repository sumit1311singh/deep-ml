
def OSA(source: str, target: str) -> int:
    m, n = len(source), len(target)

    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: insertions and deletions
    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):

            cost = 0 if source[i - 1] == target[j - 1] else 1

            # Insert, delete, or substitute
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )

            # Transposition of adjacent characters
            if (i > 1 and j > 1 and
                source[i - 1] == target[j - 2] and
                source[i - 2] == target[j - 1]):

                dp[i][j] = min(
                    dp[i][j],
                    dp[i - 2][j - 2] + 1
                )

    return dp[m][n]
