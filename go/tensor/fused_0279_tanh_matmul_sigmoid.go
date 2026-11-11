package tensor

import "fmt"

func Fused0279TanhMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Sigmoid()
	return r, nil
}
func Fused0279TanhMatmulSigmoidName() string { return "tanh_matmul_sigmoid" }
