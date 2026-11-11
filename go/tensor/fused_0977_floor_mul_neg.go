package tensor

import "fmt"

func Fused0977FloorMulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_neg: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Neg()
	return r, nil
}
func Fused0977FloorMulNegName() string { return "floor_mul_neg" }
