package tensor

import "fmt"

func Fused0452SqrtMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Sqrt()
	r = r.ReLU()
	return r, nil
}
func Fused0452SqrtMulReluName() string { return "sqrt_mul_relu" }
