package tensor

import "fmt"

func Fused1507NegMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.GELU()
	return r, nil
}
func Fused1507NegMulGeluName() string { return "neg_mul_gelu" }
