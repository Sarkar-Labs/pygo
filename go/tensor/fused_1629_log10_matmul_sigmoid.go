package tensor

import "fmt"

func Fused1629Log10MatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Sigmoid()
	return r, nil
}
func Fused1629Log10MatmulSigmoidName() string { return "log10_matmul_sigmoid" }
