package tensor

import "fmt"

func Fused1194SignMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Sqrt()
	return r, nil
}
func Fused1194SignMatmulSqrtName() string { return "sign_matmul_sqrt" }
