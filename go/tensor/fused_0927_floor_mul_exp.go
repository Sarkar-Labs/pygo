package tensor

import "fmt"

func Fused0927FloorMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Exp()
	return r, nil
}
func Fused0927FloorMulExpName() string { return "floor_mul_exp" }
