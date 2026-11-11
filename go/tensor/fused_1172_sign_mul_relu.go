package tensor

import "fmt"

func Fused1172SignMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sign()
	r = r.ReLU()
	return r, nil
}
func Fused1172SignMulReluName() string { return "sign_mul_relu" }
