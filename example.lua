nn = require("nn")
X = {{0,0,1},{0,1,1},{1,0,1},{1,1,1}}
--y should be a 2 layer table (fix subtract)
y = {0, 0, 1, 1}
l1 = nn.new(X,y)
l1:train(10000)
local z = l1:predict(X)
for i=1,#z do
    print(z[i])
end
print(l1:predict(X[1]))
