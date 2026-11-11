package tensor

import "fmt"

func Fused0812CosMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.ReLU()
	return r, nil
}
func Fused0812CosMulReluName() string { return "cos_mul_relu" }
