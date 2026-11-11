package tensor

import "fmt"

func Fused1449NegMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.Sigmoid()
	return r, nil
}
func Fused1449NegMatmulSigmoidName() string { return "neg_matmul_sigmoid" }
