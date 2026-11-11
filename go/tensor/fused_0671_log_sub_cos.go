package tensor

import "fmt"

func Fused0671LogSubCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_sub_cos: nil operand")
	}
	r := x.Sub(y)
	r = r.Log()
	r = r.Cos()
	return r, nil
}
func Fused0671LogSubCosName() string { return "log_sub_cos" }
