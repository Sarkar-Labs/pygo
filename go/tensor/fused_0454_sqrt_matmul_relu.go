package tensor

import "fmt"

func Fused0454SqrtMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.ReLU()
	return r, nil
}
func Fused0454SqrtMatmulReluName() string { return "sqrt_matmul_relu" }
