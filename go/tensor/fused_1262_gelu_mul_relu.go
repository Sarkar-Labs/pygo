package tensor

import "fmt"

func Fused1262GeluMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.GELU()
	r = r.ReLU()
	return r, nil
}
func Fused1262GeluMulReluName() string { return "gelu_mul_relu" }
