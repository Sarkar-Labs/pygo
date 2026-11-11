package tensor

import "fmt"

func Fused0814CosMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.ReLU()
	return r, nil
}
func Fused0814CosMatmulReluName() string { return "cos_matmul_relu" }
