package tensor

import "fmt"

func Fused1444NegMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Neg()
	r = r.ReLU()
	return r, nil
}
func Fused1444NegMatmulReluName() string { return "neg_matmul_relu" }
