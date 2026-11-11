package tensor

import "fmt"

func Fused1482NegMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Cos()
	return r, nil
}
func Fused1482NegMulCosName() string { return "neg_mul_cos" }
