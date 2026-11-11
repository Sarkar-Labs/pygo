package tensor

import "fmt"

func Fused0364AbsMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.ReLU()
	return r, nil
}
func Fused0364AbsMatmulReluName() string { return "abs_matmul_relu" }
