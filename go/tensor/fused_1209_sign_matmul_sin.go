package tensor

import "fmt"

func Fused1209SignMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Sin()
	return r, nil
}
func Fused1209SignMatmulSinName() string { return "sign_matmul_sin" }
