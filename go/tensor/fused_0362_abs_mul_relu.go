package tensor

import "fmt"

func Fused0362AbsMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.ReLU()
	return r, nil
}
func Fused0362AbsMulReluName() string { return "abs_mul_relu" }
