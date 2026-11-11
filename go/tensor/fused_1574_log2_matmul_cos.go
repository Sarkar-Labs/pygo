package tensor

import "fmt"

func Fused1574Log2MatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Cos()
	return r, nil
}
func Fused1574Log2MatmulCosName() string { return "log2_matmul_cos" }
