package tensor

import "fmt"

func Fused0272TanhMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Tanh()
	r = r.ReLU()
	return r, nil
}
func Fused0272TanhMulReluName() string { return "tanh_mul_relu" }
