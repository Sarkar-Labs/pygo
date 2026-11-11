package tensor

import "fmt"

func Fused1532Log2MulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.ReLU()
	return r, nil
}
func Fused1532Log2MulReluName() string { return "log2_mul_relu" }
