package tensor

import "fmt"

func Fused0169ReluMatmulNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_matmul_neg: nil operand")
	}
	r := x.MatMul(y)
	r = r.ReLU()
	r = r.Neg()
	return r, nil
}
func Fused0169ReluMatmulNegName() string { return "relu_matmul_neg" }
