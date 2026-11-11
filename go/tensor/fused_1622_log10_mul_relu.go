package tensor

import "fmt"

func Fused1622Log10MulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Log10()
	r = r.ReLU()
	return r, nil
}
func Fused1622Log10MulReluName() string { return "log10_mul_relu" }
