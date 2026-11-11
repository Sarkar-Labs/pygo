package tensor

import "fmt"

func Fused0047FloorMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	return r, nil
}
func Fused0047FloorMulName() string { return "floor_mul" }
