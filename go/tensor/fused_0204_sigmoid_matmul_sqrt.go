package tensor

import "fmt"

func Fused0204SigmoidMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Sqrt()
	return r, nil
}
func Fused0204SigmoidMatmulSqrtName() string { return "sigmoid_matmul_sqrt" }
