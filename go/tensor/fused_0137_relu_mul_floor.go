package tensor

import "fmt"

func Fused0137ReluMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Floor()
	return r, nil
}
func Fused0137ReluMulFloorName() string { return "relu_mul_floor" }
