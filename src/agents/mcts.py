# MCTS control flow
# while searching:
#     rollout
#     leaf batch append leaf
#     if leaf batch enough:
#         isend leaf batch
#     while value batch exist:
#         recv value batch
#     Adjust Tree weights
# Return value, move tensor
