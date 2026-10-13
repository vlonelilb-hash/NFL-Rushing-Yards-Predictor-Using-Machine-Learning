from sklearn.linear_model import LinearRegression

# Previous games: carries, rushing yards, opponent defense
X = [
    [15, 75, 100],
    [20, 90, 110],
    [12, 48, 95],
    [18, 108, 120],
    [22, 88, 105],
    [16, 80, 115],
    [25, 125, 130],
    [14, 56, 90],
    [19, 95, 100],
    [21, 105, 125],
    [17, 68, 110],
    [24, 120, 115]
]

# Actual rushing yards in those games
y = [75, 90, 48, 108, 88, 80, 125, 56, 95, 105, 68, 120]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Upcoming game predictions
last_3_carries = [18, 20, 16]
last_3_yards = [85, 95, 72]

average_carries = sum(last_3_carries) / 3
average_yards = sum(last_3_yards) / 3

# Example opponent defense rating
opponent_defense = 110

# Predict rushing yards before the game
prediction = model.predict([[
    average_carries,
    average_yards,
    opponent_defense
]])

print("Average carries:", round(average_carries, 1))
print("Average rushing yards:", round(average_yards, 1))
print("Predicted rushing yards:", round(prediction[0], 1))
