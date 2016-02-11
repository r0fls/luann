-- sigmoid function
function nonlin(x, deriv)
    deriv = deriv or false
    if deriv then
        return nonlin(x)*(1-nonlin(x))
    end
    return 1/(1+math.exp(-x))
end

function nonlin_array(x, deriv)
    local res = {}
    for i = 1, #x do
        res[i] = nonlin(x[i], deriv)
    end
    return res
end

local X = {{0,0,1},{0,1,1},{1,0,1},{1,1,1}}

local y = {0, 0, 1, 1}

--weights (should be a function)
function weights(len)
    local w = {}
    for i=1,len do
        w[i] = 2*math.random()-1
    end
    return w
end

w = weights(3)

function dotprod(a, b)
    local res = 0
    for i = 1, #a do
        res = res + a[i] * b[i]
    end
    return res
end

function matrixdot(a, b)
    local res = {}
    for i = 1, #a do
        res[i] = dotprod(a[i],b)
    end
    return res
end

function multiply(a, b)
    local res = {}
    for i=1,#a do
        res[i]=a[i]*b[i]
    end
    return res
end
--subtract two equally sized tables; returns a table
function subtract(a, b)
    local res = {}
    for i=1,#a do
        res[i]=a[i]-b[i]
    end
    return res
end
--add two tables; consolidate with above
function add(a, b)
    local res = {}
    for i=1,#a do
        res[i]=a[i]+b[i]
    end
    return res
end

local nn = {}

function nn.train(iterations,input,output,w)
    if not w then
        w = weights(#output)
    end
    for i = 1,iterations do
        local l0 = input
        l1 = nonlin_array(matrixdot(l0, w))
        l1_error = subtract(output,l1)
        l1_delta = multiply(l1_error, nonlin_array(l1, true))
        w = add(w, matrixdot(l0, l1_delta))
    end
    return l1
end

local l1 = nn.train(10000,X,y)

for i=1,#l1 do
    print(l1[i])
end
