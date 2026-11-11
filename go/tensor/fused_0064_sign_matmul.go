package tensor

import "fmt"

func Fused0064SignMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	return r, nil
}
func Fused0064SignMatmulName() string { return "sign_matmul" }
