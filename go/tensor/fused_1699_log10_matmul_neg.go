package tensor

import "fmt"

func Fused1699Log10MatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Neg()
	return r, nil
}
func Fused1699Log10MatmulNegName() string { return "log10_matmul_neg" }
