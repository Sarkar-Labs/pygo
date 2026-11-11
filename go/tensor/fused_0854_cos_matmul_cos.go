package tensor

import "fmt"

func Fused0854CosMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Cos()
	return r, nil
}
func Fused0854CosMatmulCosName() string { return "cos_matmul_cos" }
