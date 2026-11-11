package tensor

import "fmt"

func Fused1457NegMulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_abs: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Abs()
	return r, nil
}
func Fused1457NegMulAbsName() string { return "neg_mul_abs" }
