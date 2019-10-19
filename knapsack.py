def get_knapsack_v_w(n):
    v_w = list()
    for j in range(int(n)):
        # Get Value and Weight in one line separated by space
        v_w.append(list(map(int, input().split())))

    return v_w
