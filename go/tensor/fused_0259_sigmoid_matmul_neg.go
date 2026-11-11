package tensor

import "fmt"

func Fused0259SigmoidMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Neg()
	return r, nil
}
func Fused0259SigmoidMatmulNegName() string { return "sigmoid_matmul_neg" }
