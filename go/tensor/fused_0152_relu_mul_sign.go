package tensor

import "fmt"

func Fused0152ReluMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Sign()
	return r, nil
}
func Fused0152ReluMulSignName() string { return "relu_mul_sign" }
