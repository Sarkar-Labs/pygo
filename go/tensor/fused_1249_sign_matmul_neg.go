package tensor

import "fmt"

func Fused1249SignMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Neg()
	return r, nil
}
func Fused1249SignMatmulNegName() string { return "sign_matmul_neg" }
