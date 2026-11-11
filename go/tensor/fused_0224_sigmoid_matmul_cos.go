package tensor

import "fmt"

func Fused0224SigmoidMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Cos()
	return r, nil
}
func Fused0224SigmoidMatmulCosName() string { return "sigmoid_matmul_cos" }
