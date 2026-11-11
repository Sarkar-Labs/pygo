package tensor

import "fmt"

func Fused0917FloorMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Abs()
	return r, nil
}
func Fused0917FloorMulAbsName() string { return "floor_mul_abs" }
