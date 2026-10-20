-- Global stability of a small network: any two valid inputs that are at most
-- epsilon apart (in every coordinate) have outputs at most delta apart.
-- Both network applications have free inputs, so this needs the product
-- network and change of variables in vnnlib-abcrown.

type Input = Tensor Real [2]

@network
toy : Input -> Tensor Real [2]

@parameter
epsilon : Real

@parameter
delta : Real

validInput : Input -> Bool
validInput x = forall i . 0 <= x ! i <= 1

close : Input -> Input -> Bool
close x y = forall i . -epsilon <= x ! i - y ! i <= epsilon

@property
globallyStable : Bool
globallyStable = forall x y .
  validInput x and validInput y and close x y =>
    (forall i . -delta <= toy x ! i - toy y ! i <= delta)
