package tensor

import "fmt"

func Fused1224SignMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Ceil()
	return r, nil
}
func Fused1224SignMatmulCeilName() string { return "sign_matmul_ceil" }
