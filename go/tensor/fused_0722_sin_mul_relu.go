package tensor

import "fmt"

func Fused0722SinMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sin()
	r = r.ReLU()
	return r, nil
}
func Fused0722SinMulReluName() string { return "sin_mul_relu" }
