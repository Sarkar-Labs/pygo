package tensor

import "fmt"

func Fused0157ReluMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.GELU()
	return r, nil
}
func Fused0157ReluMulGeluName() string { return "relu_mul_gelu" }
