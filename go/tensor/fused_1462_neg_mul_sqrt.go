package tensor

import "fmt"

func Fused1462NegMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Sqrt()
	return r, nil
}
func Fused1462NegMulSqrtName() string { return "neg_mul_sqrt" }
