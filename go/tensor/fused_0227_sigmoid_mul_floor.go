package tensor

import "fmt"

func Fused0227SigmoidMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Sigmoid()
	r = r.Floor()
	return r, nil
}
func Fused0227SigmoidMulFloorName() string { return "sigmoid_mul_floor" }
