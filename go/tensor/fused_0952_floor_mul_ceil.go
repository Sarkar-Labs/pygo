package tensor

import "fmt"

func Fused0952FloorMulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_ceil: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Ceil()
	return r, nil
}
func Fused0952FloorMulCeilName() string { return "floor_mul_ceil" }
