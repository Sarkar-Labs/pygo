package tensor

import "fmt"

func Fused0264SigmoidMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Log2()
	return r, nil
}
func Fused0264SigmoidMatmulLog2Name() string { return "sigmoid_matmul_log2" }
