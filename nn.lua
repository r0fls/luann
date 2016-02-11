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
nn.__index = nn

function nn.new(input, output, w)
    local self = setmetatable({}, nn)
    if not w then
        w = weights(#output)
    end
    self.w = w
    self.input = input
    self.output = output
    return self 
end

function nn.train(self, iterations)
    for i = 1,iterations do
        local l0 = self.input
        local l1 = nonlin_array(matrixdot(l0, self.w))
        l1_error = subtract(self.output,l1)
        l1_delta = multiply(l1_error, nonlin_array(l1, true))
        self.w = add(self.w, matrixdot(l0, l1_delta))
    end
    return self
end

function nn.predict(self, input)
    if type(input[1])=='table' then
        return nonlin_array(matrixdot(input, self.w))
    elseif type(input[1])=='number' then
        return nonlin(dotprod(input, self.w))
    end
end

return nn
