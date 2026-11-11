package tensor

import "fmt"

func Fused0907FloorMulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_sigmoid: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Sigmoid()
	return r, nil
}
func Fused0907FloorMulSigmoidName() string { return "floor_mul_sigmoid" }
